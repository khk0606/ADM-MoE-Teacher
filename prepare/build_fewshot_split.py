#!/usr/bin/env python3
"""Build leakage-free CDM and MoE-IIW splits from Unity exports.

The split unit is the connected component of raw motion sources, not the
exported sample directory.  Consequently, a multi-start derivative can never
land in a different split from its parent/original motion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Set


class DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def read_json(path: Path) -> MutableMapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def stable_digest(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def production_sample(sample_dir: Path) -> Optional[Dict[str, object]]:
    manifest_path = sample_dir / "sample_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = read_json(manifest_path)
    sample_id = str(manifest.get("sample_id", sample_dir.name))
    scene_id = str(manifest.get("scene_id", ""))
    target = str(manifest.get("target_name", "")).lower()
    text = str(manifest.get("text", ""))

    if scene_id in {"room_0001", "room_0002"} and target == "chair":
        return {
            "sample_id": sample_id,
            "scene_id": scene_id,
            "target": target,
            "text": text,
            "stage": "base",
            "original_motion_id": sample_id,
            "source_components": [f"legacy:{sample_id}"],
            "parent_base_id": "",
        }

    # room_0003/0004 sample 0001 files are the explicit-object diagnostics.
    if sample_id in {
        "room_0003_lie_bed_0001",
        "room_0004_interact_whiteboard_0001",
    }:
        return None
    if (scene_id, target) not in {
        ("room_0003", "bed"),
        ("room_0004", "whiteboard"),
    }:
        return None

    sidecar_path = sample_dir / "derivation_manifest.json"
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"{sidecar_path}: new Bed/Whiteboard samples require provenance"
        )
    sidecar = read_json(sidecar_path)
    if target in text.lower():
        raise ValueError(
            f"{manifest_path}: production prompt leaks target word {target!r}"
        )
    components = [str(value) for value in sidecar.get("source_component_ids", [])]
    original = str(sidecar.get("original_motion_id", ""))
    if not original or not components:
        raise ValueError(f"{sidecar_path}: incomplete source provenance")
    return {
        "sample_id": sample_id,
        "scene_id": scene_id,
        "target": target,
        "text": text,
        "stage": str(sidecar.get("stage", "")),
        "original_motion_id": original,
        "source_components": components,
        "parent_base_id": str(sidecar.get("parent_base_id", "")),
    }


def assign_group_splits(
    groups: Sequence[Dict[str, object]], test_fraction: float, seed: int
) -> Dict[str, str]:
    by_target: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for group in groups:
        targets = list(group["targets"])
        if len(targets) != 1:
            raise ValueError(
                f"source group {group['group_id']} spans targets: {targets}"
            )
        by_target[str(targets[0])].append(group)

    assignment: Dict[str, str] = {}
    for target, target_groups in sorted(by_target.items()):
        if len(target_groups) < 2:
            raise ValueError(
                f"target {target!r} has only {len(target_groups)} independent "
                "source group(s); strict train/test evaluation is impossible"
            )
        ordered = sorted(
            target_groups,
            key=lambda group: stable_digest(str(group["group_id"]), seed),
        )
        test_count = max(1, round(len(ordered) * test_fraction))
        test_count = min(test_count, len(ordered) - 1)
        test_ids = {str(group["group_id"]) for group in ordered[:test_count]}
        for group in ordered:
            group_id = str(group["group_id"])
            assignment[group_id] = "test" if group_id in test_ids else "train"
    return assignment


def assert_disjoint(samples: Sequence[Dict[str, object]]) -> None:
    split_components: Dict[str, Set[str]] = defaultdict(set)
    split_originals: Dict[str, Set[str]] = defaultdict(set)
    for sample in samples:
        split = str(sample["split"])
        split_components[split].update(str(v) for v in sample["source_components"])
        split_originals[split].add(str(sample["original_motion_id"]))
    component_overlap = split_components["train"] & split_components["test"]
    original_overlap = split_originals["train"] & split_originals["test"]
    if component_overlap or original_overlap:
        raise AssertionError(
            "source leakage detected: components="
            f"{sorted(component_overlap)}, originals={sorted(original_overlap)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unity-root",
        type=Path,
        default=Path("data/unity_exports/history_affordance_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/history_affordance_v1/splits/"
            "chair23_bed2_whiteboard12_multistart24_v1.json"
        ),
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("--test-fraction must be in (0, 1)")

    sample_root = args.unity_root.resolve() / "samples"
    samples = []
    for sample_dir in sorted(sample_root.iterdir()):
        if sample_dir.is_dir():
            sample = production_sample(sample_dir)
            if sample is not None:
                samples.append(sample)
    if not samples:
        raise RuntimeError(f"{sample_root}: no production samples")

    nodes: Set[str] = set()
    for sample in samples:
        nodes.add("original:" + str(sample["original_motion_id"]))
        nodes.update("component:" + str(v) for v in sample["source_components"])
    dsu = DisjointSet(nodes)
    for sample in samples:
        sample_nodes = ["original:" + str(sample["original_motion_id"])]
        sample_nodes.extend("component:" + str(v) for v in sample["source_components"])
        for node in sample_nodes[1:]:
            dsu.union(sample_nodes[0], node)

    root_samples: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for sample in samples:
        node = "original:" + str(sample["original_motion_id"])
        root_samples[dsu.find(node)].append(sample)

    groups = []
    for root, members in root_samples.items():
        component_values = sorted(
            {
                str(component)
                for sample in members
                for component in sample["source_components"]
            }
        )
        originals = sorted({str(v["original_motion_id"]) for v in members})
        group_id = hashlib.sha256(
            "\n".join(component_values + originals).encode("utf-8")
        ).hexdigest()[:16]
        groups.append(
            {
                "group_id": group_id,
                "root": root,
                "targets": sorted({str(v["target"]) for v in members}),
                "source_components": component_values,
                "original_motion_ids": originals,
                "sample_ids": sorted(str(v["sample_id"]) for v in members),
            }
        )
    assignment = assign_group_splits(groups, args.test_fraction, args.seed)
    sample_to_group = {
        sample_id: str(group["group_id"])
        for group in groups
        for sample_id in group["sample_ids"]
    }
    for sample in samples:
        group_id = sample_to_group[str(sample["sample_id"])]
        sample["source_group_id"] = group_id
        sample["split"] = assignment[group_id]
    assert_disjoint(samples)

    cdm = {"train": [], "test": []}
    iiw = {"train": [], "test": []}
    for sample in samples:
        split = str(sample["split"])
        sample_id = str(sample["sample_id"])
        iiw[split].append(sample_id)
        if sample["stage"] == "base":
            cdm[split].append(sample_id)

    expected = {
        ("chair", "base"): 23,
        ("bed", "base"): 2,
        ("whiteboard", "base"): 12,
        ("whiteboard", "multistart"): 24,
    }
    observed = Counter((str(v["target"]), str(v["stage"])) for v in samples)
    if dict(observed) != expected:
        raise AssertionError(f"sample contract mismatch: {dict(observed)} != {expected}")

    for group in groups:
        group["split"] = assignment[str(group["group_id"])]
        group.pop("root", None)
    payload = {
        "schema": "affordance_source_disjoint_split_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "split_unit": "connected raw-source component",
        "diagnostic_samples_excluded": [
            "room_0003_lie_bed_0001",
            "room_0004_interact_whiteboard_0001",
        ],
        "cdm_fewshot": {key: sorted(value) for key, value in cdm.items()},
        "moe_iiw": {key: sorted(value) for key, value in iiw.items()},
        "groups": sorted(groups, key=lambda value: str(value["group_id"])),
        "samples": sorted(samples, key=lambda value: str(value["sample_id"])),
        "checks": {
            "source_components_disjoint": True,
            "original_motion_ids_disjoint": True,
            "multistart_excluded_from_cdm": True,
            "object_names_absent_from_new_prompts": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"[PASS] source-disjoint split: {len(groups)} groups, {len(samples)} samples")
    print(f"[OK] CDM train/test: {len(cdm['train'])}/{len(cdm['test'])}")
    print(f"[OK] IIW train/test: {len(iiw['train'])}/{len(iiw['test'])}")
    for target in ("chair", "bed", "whiteboard"):
        counts = Counter(
            str(sample["split"]) for sample in samples if sample["target"] == target
        )
        print(f"[OK] {target}: train={counts['train']} test={counts['test']}")
    print(f"[OK] saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()

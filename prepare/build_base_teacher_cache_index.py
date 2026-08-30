#!/usr/bin/env python3
"""Bind promoted Base-teacher artifacts to all MoE-IIW samples.

This step performs no model sampling. It validates four immutable scene/prompt
artifacts, the Gate 0C promotion report, the source-disjoint split, and the
past-only history index before atomically writing one sample-to-cache index.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple


PREPARE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PREPARE_DIR.parent
if not (REPO_ROOT / "history_affordance_v2_moe_patch").is_dir():
    # Source-patch layout on the packaging host. On Ubuntu this file is
    # installed directly under ~/AMDM/prepare and the first candidate wins.
    REPO_ROOT = REPO_ROOT.parent
for path in (REPO_ROOT, PREPARE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from base_teacher_contract import (  # noqa: E402
    CHANNEL_JOINT_INDICES,
    CHANNEL_ORDER,
    PROMPT_BY_TARGET,
    PROMPT_POLICY_ID,
    canonical_json_sha256,
    load_json,
    load_scene_contract,
    sha256_file,
    validate_base_teacher_artifact,
)
from history_affordance_v2_moe_patch.datasets.past_history_store import (  # noqa: E402
    PastHistoryWindowStore,
)


SCHEMA = "history_affordance_v2_base_teacher_cache_index_v1"
SPLIT_SCHEMA = "affordance_source_disjoint_split_v1"
HISTORY_SCHEMA = "history_affordance_past_window_index_v1"
PREFLIGHT_SCHEMA = "history_affordance_v2_base_teacher_preflight_v1"
PROMPT_ID_BY_TARGET = {
    "chair": "sit_generic_v1",
    "bed": "lie_generic_v1",
    "whiteboard": "write_right_hand_generic_v1",
}
SHA_PATTERN = re.compile(r"[0-9a-f]{64}")


def _relative_to_dataset(path: Path, dataset_root: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(dataset_root))
    except ValueError as error:
        raise ValueError(str(resolved) + " is outside dataset root") from error


def _split_rows(split_file: Path) -> Tuple[Dict[str, Dict[str, object]], Sequence[str]]:
    split = load_json(split_file)
    if split.get("schema") != SPLIT_SCHEMA:
        raise ValueError("source-disjoint split schema mismatch")
    checks = split.get("checks")
    if not isinstance(checks, Mapping) or not checks or not all(
        value is True for value in checks.values()
    ):
        raise ValueError("source-disjoint split checks are not all true")
    moe = split.get("moe_iiw")
    if not isinstance(moe, Mapping):
        raise TypeError("split.moe_iiw must be a mapping")
    train = moe.get("train")
    test = moe.get("test")
    if not isinstance(train, list) or not isinstance(test, list):
        raise TypeError("split MoE train/test lists are missing")
    selected = [str(value) for value in train + test]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("MoE split sample IDs must be non-empty and unique")
    rows = split.get("samples")
    if not isinstance(rows, list):
        raise TypeError("split.samples must be a list")
    by_id: Dict[str, Dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("split sample row must be a mapping")
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in by_id:
            raise ValueError("split sample IDs must be non-empty and unique")
        by_id[sample_id] = row
    if set(by_id) != set(selected):
        raise ValueError("split sample rows differ from MoE train/test IDs")
    for sample_id in train:
        if by_id[str(sample_id)].get("split") != "train":
            raise ValueError(str(sample_id) + ": train split declaration mismatch")
    for sample_id in test:
        if by_id[str(sample_id)].get("split") != "test":
            raise ValueError(str(sample_id) + ": test split declaration mismatch")
    return by_id, sorted(selected)


def expected_pairs(
    split_rows: Mapping[str, Mapping[str, object]],
) -> Dict[Tuple[str, str, str], Sequence[str]]:
    pairs: Dict[Tuple[str, str, str], list] = {}
    for sample_id, row in sorted(split_rows.items()):
        target = str(row.get("target", ""))
        if target not in PROMPT_BY_TARGET or target not in PROMPT_ID_BY_TARGET:
            raise ValueError(sample_id + ": unsupported target " + target)
        scene_id = str(row.get("scene_id", ""))
        if not scene_id:
            raise ValueError(sample_id + ": scene_id is empty")
        key = (
            scene_id,
            PROMPT_ID_BY_TARGET[target],
            PROMPT_BY_TARGET[target],
        )
        pairs.setdefault(key, []).append(sample_id)
    return {key: tuple(value) for key, value in sorted(pairs.items())}


def _validate_promotion_report(
    report_file: Path,
    *,
    canary_cache_key: str,
    canary_result: Mapping[str, object],
    canary_manifest: Mapping[str, object],
) -> Dict[str, object]:
    report = load_json(report_file)
    required_top = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "PASS",
        "promotion_authorized": True,
        "mode": "full",
        "cache_key": canary_cache_key,
        "manifest_sha256": canary_result["manifest_sha256"],
        "artifact_id": canary_result["artifact_id"],
        "artifact_sha256": canary_result["artifact_sha256"],
    }
    mismatched = {
        key: {"expected": value, "actual": report.get(key)}
        for key, value in required_top.items()
        if report.get(key) != value
    }
    if mismatched:
        raise ValueError("Gate 0C report/canary mismatch: " + str(mismatched))
    checks = report.get("checks")
    if not isinstance(checks, Mapping) or not checks or not all(
        value is True for value in checks.values()
    ):
        raise ValueError("Gate 0C common checks are not all true")
    replay = report.get("full_replay")
    if not isinstance(replay, Mapping):
        raise TypeError("Gate 0C full_replay is missing")
    replay_requirements = {
        "status": "PASS",
        "promotion_authorized": True,
        "all_draws_reproduced_bitwise": True,
        "draw_zero_repeatability_bitwise": True,
        "quality_prediction_all_reproduced_bitwise": True,
        "quality_prediction_total_replayed_draws": 370,
        "sampling_calls_preserved_caller_rng_state": True,
        "state_unchanged_during_preflight": True,
        "teacher_runtime_sha256": canary_manifest["runtime_provenance"][
            "teacher_runtime_sha256"
        ],
    }
    replay_mismatched = {
        key: {"expected": value, "actual": replay.get(key)}
        for key, value in replay_requirements.items()
        if replay.get(key) != value
    }
    if replay_mismatched:
        raise ValueError("Gate 0C replay contract mismatch: " + str(replay_mismatched))
    quality = report.get("quality_chain")
    if not isinstance(quality, Mapping) or quality.get("status") != "PASS":
        raise ValueError("Gate 0C quality chain is not PASS")
    return report


def _cache_index_id(index: Mapping[str, object]) -> str:
    payload = dict(index)
    payload.pop("index_id", None)
    return canonical_json_sha256(payload)


def validate_cache_index(
    index_file: Path,
    *,
    verify_history_sources: bool = True,
) -> Dict[str, object]:
    index_file = index_file.expanduser().resolve()
    index = load_json(index_file)
    if index.get("schema") != SCHEMA or index.get("status") != "PASS":
        raise ValueError("Base cache index schema/status mismatch")
    if index.get("promotion_authorized") is not True:
        raise ValueError("Base cache index is not promotion-authorized")
    if index.get("prompt_policy_id") != PROMPT_POLICY_ID:
        raise ValueError("Base cache prompt policy mismatch")
    if index.get("channel_order") != list(CHANNEL_ORDER) or index.get(
        "channel_joint_indices"
    ) != list(CHANNEL_JOINT_INDICES):
        raise ValueError("Base cache channel contract mismatch")
    if index.get("index_id") != _cache_index_id(index):
        raise ValueError("Base cache index ID mismatch")
    dataset_root = Path(str(index["dataset_root"])).expanduser().resolve()
    split_file = dataset_root / str(index["split_file"])
    history_file = dataset_root / str(index["history_index"])
    preflight_file = dataset_root / str(index["gate0_full_preflight"])
    for path, recorded in (
        (split_file, index["split_file_sha256"]),
        (history_file, index["history_index_sha256"]),
        (preflight_file, index["gate0_full_preflight_sha256"]),
    ):
        if not path.is_file() or sha256_file(path) != recorded:
            raise ValueError(str(path) + ": bound source SHA-256 mismatch")
    history = PastHistoryWindowStore(
        history_file, verify_source_files=verify_history_sources
    )
    pairs = index.get("pairs")
    samples = index.get("samples")
    if not isinstance(pairs, list) or not isinstance(samples, list):
        raise TypeError("Base cache pairs/samples must be lists")
    if len(pairs) != int(index["num_pairs"]):
        raise ValueError("Base cache pair count mismatch")
    if len(samples) != int(index["num_samples"]):
        raise ValueError("Base cache sample count mismatch")
    pair_by_key = {}
    for pair in pairs:
        if not isinstance(pair, dict):
            raise TypeError("Base cache pair row must be a mapping")
        key = str(pair.get("cache_key", ""))
        if key in pair_by_key:
            raise ValueError("duplicate Base cache key")
        manifest_file = dataset_root / str(pair["manifest_file"])
        artifact_file = dataset_root / str(pair["artifact_file"])
        result = validate_base_teacher_artifact(
            manifest_file=manifest_file, artifact_file=artifact_file
        )
        for name in (
            "cache_key",
            "manifest_sha256",
            "artifact_sha256",
            "artifact_id",
            "scene_id",
            "prompt_id",
        ):
            if pair.get(name) != result.get(name):
                raise ValueError(key + ": pair/validated artifact " + name + " mismatch")
        pair_by_key[key] = pair
    sample_ids = []
    for row in samples:
        if not isinstance(row, dict):
            raise TypeError("Base cache sample row must be a mapping")
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in sample_ids:
            raise ValueError("Base cache sample IDs must be non-empty and unique")
        sample_ids.append(sample_id)
        if str(row.get("cache_key", "")) not in pair_by_key:
            raise ValueError(sample_id + ": missing Base cache pair")
    if sorted(sample_ids) != sorted(history.sample_ids):
        raise ValueError("Base cache and history sample IDs differ")
    checks = index.get("checks")
    if not isinstance(checks, Mapping) or not checks or not all(
        value is True for value in checks.values()
    ):
        raise ValueError("Base cache index checks are not all true")
    return {
        "status": "PASS",
        "index": str(index_file),
        "index_id": str(index["index_id"]),
        "num_pairs": len(pairs),
        "num_samples": len(samples),
        "history_frames": history.history_frames,
        "history_feature_dim": history.feature_dim,
        "all_artifact_hashes_verified": True,
        "all_history_hashes_verified": True,
        "promotion_report_verified": True,
    }


def build_cache_index(
    *,
    dataset_root: Path,
    split_file: Path,
    history_index: Path,
    artifact_root: Path,
    cache_keys: Sequence[str],
    canary_cache_key: str,
    full_preflight: Path,
    output_dir: Path,
) -> Path:
    dataset_root = dataset_root.expanduser().resolve()
    split_file = split_file.expanduser().resolve()
    history_index = history_index.expanduser().resolve()
    artifact_root = artifact_root.expanduser().resolve()
    full_preflight = full_preflight.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    for path in (dataset_root, split_file, history_index, artifact_root, full_preflight):
        if not path.exists():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite Base cache index: " + str(output_dir))
    if len(cache_keys) != len(set(cache_keys)) or any(
        SHA_PATTERN.fullmatch(str(value)) is None for value in cache_keys
    ):
        raise ValueError("cache keys must be unique 64-character lowercase SHA-256")
    if canary_cache_key not in cache_keys:
        raise ValueError("canary cache key is absent from cache-key list")

    split_rows, selected_ids = _split_rows(split_file)
    expected = expected_pairs(split_rows)
    history = PastHistoryWindowStore(history_index, verify_source_files=True)
    if sorted(history.sample_ids) != list(selected_ids):
        raise ValueError("history and source-disjoint split sample IDs differ")
    if history.history_frames != 10 or history.stride != 1:
        raise ValueError("production history contract requires H=10, stride=1")
    history_index_json = load_json(history_index)
    if history_index_json.get("schema") != HISTORY_SCHEMA:
        raise ValueError("history index schema mismatch")
    if Path(str(history_index_json["split_file"])).resolve() != split_file:
        raise ValueError("history index is bound to a different split path")
    if history_index_json.get("split_file_sha256") != sha256_file(split_file):
        raise ValueError("history/split SHA-256 mismatch")

    validated = []
    manifests = {}
    for cache_key in cache_keys:
        manifest_file = artifact_root / cache_key / "manifest.json"
        artifact_file = artifact_root / cache_key / "base_teacher.npz"
        result = validate_base_teacher_artifact(
            manifest_file=manifest_file, artifact_file=artifact_file
        )
        if result["cache_key"] != cache_key:
            raise ValueError(cache_key + ": directory/cache-key mismatch")
        manifest = load_json(manifest_file)
        scene = load_scene_contract(dataset_root, str(manifest["scene_id"]))
        if manifest.get("scene_sha256") != scene["scene_sha256"]:
            raise ValueError(cache_key + ": current scene differs from artifact")
        if manifest.get("prompt_policy_id") != PROMPT_POLICY_ID:
            raise ValueError(cache_key + ": prompt policy mismatch")
        if manifest.get("conditioning", {}).get("history_conditioned") is not False:
            raise ValueError(cache_key + ": Base artifact contains history conditioning")
        validated.append((result, manifest, manifest_file, artifact_file))
        manifests[cache_key] = manifest

    canary_tuple = next(row for row in validated if row[0]["cache_key"] == canary_cache_key)
    canary_result, canary_manifest, _, _ = canary_tuple
    preflight = _validate_promotion_report(
        full_preflight,
        canary_cache_key=canary_cache_key,
        canary_result=canary_result,
        canary_manifest=canary_manifest,
    )
    teacher_runtime_sha256 = canary_manifest["runtime_provenance"][
        "teacher_runtime_sha256"
    ]
    evidence_set_sha256 = canary_manifest["quality_gate"]["evidence_set_sha256"]
    quality_sha256 = canary_manifest["quality_gate"]["sha256"]
    for result, manifest, _, _ in validated:
        cache_key = result["cache_key"]
        if manifest["runtime_provenance"]["teacher_runtime_sha256"] != teacher_runtime_sha256:
            raise ValueError(cache_key + ": teacher runtime differs from canary")
        if manifest["quality_gate"]["evidence_set_sha256"] != evidence_set_sha256:
            raise ValueError(cache_key + ": evidence-set seal differs from canary")
        if manifest["quality_gate"]["sha256"] != quality_sha256:
            raise ValueError(cache_key + ": quality-file hashes differ from canary")

    artifact_by_pair = {}
    for result, manifest, manifest_file, artifact_file in validated:
        pair_key = (
            str(manifest["scene_id"]),
            str(manifest["prompt_id"]),
            str(manifest["text"]),
        )
        if pair_key in artifact_by_pair:
            raise ValueError("duplicate artifact for scene/prompt pair: " + str(pair_key))
        artifact_by_pair[pair_key] = (result, manifest, manifest_file, artifact_file)
    if set(artifact_by_pair) != set(expected):
        raise ValueError(
            "artifact/expected scene-prompt pairs differ: expected={} actual={}".format(
                sorted(expected), sorted(artifact_by_pair)
            )
        )

    pair_rows = []
    sample_rows = []
    for pair_key, member_ids in expected.items():
        result, manifest, manifest_file, artifact_file = artifact_by_pair[pair_key]
        targets = {str(split_rows[sample_id]["target"]) for sample_id in member_ids}
        if len(targets) != 1:
            raise ValueError("one scene/prompt pair spans multiple targets")
        split_counts = Counter(str(split_rows[sample_id]["split"]) for sample_id in member_ids)
        stage_counts = Counter(str(split_rows[sample_id].get("stage", "")) for sample_id in member_ids)
        pair_rows.append(
            {
                "cache_key": result["cache_key"],
                "artifact_id": result["artifact_id"],
                "scene_id": result["scene_id"],
                "target": next(iter(targets)),
                "prompt_id": result["prompt_id"],
                "text": manifest["text"],
                "manifest_file": _relative_to_dataset(manifest_file, dataset_root),
                "manifest_sha256": result["manifest_sha256"],
                "artifact_file": _relative_to_dataset(artifact_file, dataset_root),
                "artifact_sha256": result["artifact_sha256"],
                "draw_count": result["draw_count"],
                "sample_count": len(member_ids),
                "split_counts": dict(sorted(split_counts.items())),
                "stage_counts": dict(sorted(stage_counts.items())),
                "sample_ids": list(member_ids),
            }
        )
        for sample_id in member_ids:
            split_row = split_rows[sample_id]
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "scene_id": str(split_row["scene_id"]),
                    "target": str(split_row["target"]),
                    "stage": str(split_row.get("stage", "")),
                    "split": str(split_row["split"]),
                    "cache_key": result["cache_key"],
                    "prompt_id": result["prompt_id"],
                }
            )
    pair_rows.sort(key=lambda row: (row["scene_id"], row["prompt_id"]))
    sample_rows.sort(key=lambda row: row["sample_id"])
    index = {
        "schema": SCHEMA,
        "status": "PASS",
        "promotion_authorized": True,
        "dataset_root": str(dataset_root),
        "split_file": _relative_to_dataset(split_file, dataset_root),
        "split_file_sha256": sha256_file(split_file),
        "history_index": _relative_to_dataset(history_index, dataset_root),
        "history_index_sha256": sha256_file(history_index),
        "history_frames": history.history_frames,
        "history_stride": history.stride,
        "history_feature_names": list(history.feature_names),
        "gate0_full_preflight": _relative_to_dataset(full_preflight, dataset_root),
        "gate0_full_preflight_sha256": sha256_file(full_preflight),
        "gate0_canary_cache_key": canary_cache_key,
        "gate0_canary_artifact_sha256": canary_result["artifact_sha256"],
        "teacher_runtime_sha256": teacher_runtime_sha256,
        "quality_evidence_set_sha256": evidence_set_sha256,
        "prompt_policy_id": PROMPT_POLICY_ID,
        "channel_order": list(CHANNEL_ORDER),
        "channel_joint_indices": list(CHANNEL_JOINT_INDICES),
        "num_pairs": len(pair_rows),
        "num_samples": len(sample_rows),
        "num_train": sum(row["split"] == "train" for row in sample_rows),
        "num_test": sum(row["split"] == "test" for row in sample_rows),
        "pairs": pair_rows,
        "samples": sample_rows,
        "checks": {
            "gate0_full_preflight_promotion_authorized": preflight["promotion_authorized"] is True,
            "all_artifacts_integrity_pass": True,
            "all_artifacts_share_teacher_runtime": True,
            "all_artifacts_share_quality_evidence": True,
            "all_scene_prompt_pairs_exact": True,
            "all_61_samples_bound_once": len(sample_rows) == 61,
            "history_h10_s1_bound": history.history_frames == 10 and history.stride == 1,
            "base_artifacts_have_no_history_conditioning": True,
        },
    }
    if not all(index["checks"].values()):
        raise AssertionError("Base cache index contains a failed construction check")
    index["index_id"] = _cache_index_id(index)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output_dir.name + ".staging.", dir=str(output_dir.parent)))
    try:
        index_file = stage / "index.json"
        with index_file.open("w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(stage, output_dir)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    final_index = output_dir / "index.json"
    validate_cache_index(final_index, verify_history_sources=True)
    return final_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--history-index", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--cache-key", action="append", required=True)
    parser.add_argument("--canary-cache-key", required=True)
    parser.add_argument("--full-preflight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_file = build_cache_index(
        dataset_root=args.dataset_root,
        split_file=args.split,
        history_index=args.history_index,
        artifact_root=args.artifact_root,
        cache_keys=args.cache_key,
        canary_cache_key=args.canary_cache_key,
        full_preflight=args.full_preflight,
        output_dir=args.output_dir,
    )
    result = validate_cache_index(index_file, verify_history_sources=True)
    print("[PASS] promoted Base-teacher cache index")
    print("[OK] pairs: " + str(result["num_pairs"]))
    print("[OK] samples: " + str(result["num_samples"]))
    print("[OK] history: H={} D={}".format(
        result["history_frames"], result["history_feature_dim"]
    ))
    print("[OK] index ID: " + result["index_id"])
    print("[OK] saved: " + str(index_file))


if __name__ == "__main__":
    main()

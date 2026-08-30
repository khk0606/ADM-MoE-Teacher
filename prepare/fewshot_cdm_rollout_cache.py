#!/usr/bin/env python3
"""Crash-safe cache helpers for the v4 full-diffusion rollout selector.

Every cached prediction is bound to one immutable protocol contract.  A cache
is therefore reusable after Ctrl-C, SSH loss, or a Python exception, but it is
never silently reused after changing the split, checkpoints, samples, seeds,
diffusion schedule, K, or quality thresholds.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np


CONTRACT_SCHEMA = "history_affordance_v1_fewshot_cdm_rollout_cache_v1"
PREDICTION_SHAPE = (8192, 6)
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.-]+$")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(canonical_json_bytes(list(array.shape)))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def fingerprint_rows(
    rows: Mapping[str, Mapping[str, object]], ordered_ids: Sequence[str]
) -> Mapping[str, object]:
    """Hash every tensor that can affect one rollout, in deterministic order."""

    result = {}
    for sample_id in ordered_ids:
        row = rows[sample_id]
        result[sample_id] = {
            "scene_id": str(row["scene_id"]),
            "target": str(row["target"]),
            "target_instance_id": int(row["target_instance_id"]),
            "text": str(row["text"]),
            "gt_sha256": sha256_array(np.asarray(row["gt"])),
            "xyz_sha256": sha256_array(np.asarray(row["xyz"])),
            "feat_sha256": sha256_array(np.asarray(row["feat"])),
            "instance_ids_sha256": sha256_array(np.asarray(row["instance_ids"])),
            "source_indices_sha256": sha256_array(
                np.asarray(row["source_indices"])
            ),
        }
    return result


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: object) -> None:
    payload = json.dumps(
        value, indent=2, ensure_ascii=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    atomic_write_bytes(path, payload)


def atomic_save_prediction(path: Path, value: np.ndarray) -> None:
    prediction = np.asarray(value, dtype=np.float32)
    if prediction.shape != PREDICTION_SHAPE:
        raise ValueError(
            f"prediction shape {prediction.shape} != {PREDICTION_SHAPE}"
        )
    if not np.isfinite(prediction).all():
        raise ValueError("prediction contains NaN/Inf")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, prediction, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_prediction(path: Path) -> np.ndarray:
    path = Path(path)
    with path.open("rb") as handle:
        prediction = np.load(handle, allow_pickle=False)
    if prediction.dtype != np.float32:
        raise ValueError(f"{path}: cached dtype {prediction.dtype} != float32")
    if prediction.shape != PREDICTION_SHAPE:
        raise ValueError(
            f"{path}: cached shape {prediction.shape} != {PREDICTION_SHAPE}"
        )
    if not np.isfinite(prediction).all():
        raise ValueError(f"{path}: cached prediction contains NaN/Inf")
    return prediction


def _safe_token(value: str, label: str) -> str:
    if not _SAFE_TOKEN.fullmatch(value):
        raise ValueError(f"unsafe {label}: {value!r}")
    return value


def prediction_path(
    cache_dir: Path,
    role: str,
    sample_id: str,
    draw: int,
    candidate_step: Optional[int] = None,
) -> Path:
    role = _safe_token(str(role), "rollout role")
    sample_id = _safe_token(str(sample_id), "sample id")
    if draw < 0:
        raise ValueError("draw must be nonnegative")
    if role == "original":
        root = Path(cache_dir) / "predictions" / "original"
    elif role == "candidate":
        if candidate_step is None or candidate_step < 0:
            raise ValueError("candidate prediction requires a nonnegative step")
        root = (
            Path(cache_dir)
            / "predictions"
            / "candidates"
            / f"step_{candidate_step:07d}"
        )
    else:
        raise ValueError(f"unknown rollout role: {role}")
    return root / sample_id / f"draw_{draw:03d}.npy"


def candidate_result_path(cache_dir: Path, step: int) -> Path:
    if step < 0:
        raise ValueError("candidate step must be nonnegative")
    return Path(cache_dir) / "candidate_results" / f"step_{step:07d}.json"


def ensure_contract(cache_dir: Path, contract: Mapping[str, object]) -> str:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("rollout cache contract schema mismatch")
    contract_hash = sha256_json(contract)
    path = cache_dir / "contract.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != contract:
            raise RuntimeError(
                "Refusing incompatible rollout cache reuse: contract.json does "
                "not match this split/checkpoint/protocol. Use a new cache dir."
            )
        if sha256_json(existing) != contract_hash:
            raise AssertionError("rollout cache contract hash is unstable")
    else:
        atomic_write_json(path, dict(contract))
    return contract_hash


def load_bound_candidate_result(
    path: Path, contract_sha256: str, checkpoint_sha256: str, step: int
):
    path = Path(path)
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("contract_sha256") != contract_sha256:
        raise RuntimeError(f"{path}: candidate result contract mismatch")
    if value.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError(f"{path}: candidate result checkpoint mismatch")
    if int(value.get("step", -1)) != int(step):
        raise RuntimeError(f"{path}: candidate result step mismatch")
    return value

#!/usr/bin/env python3
"""Build a path-portable Teacher-v10.2 prerequisite Release asset."""

from __future__ import annotations

import argparse
import copy
import io
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Union

from teacher_v102_portable_assets import (
    ASSET_MANIFEST,
    ASSET_NAME,
    ASSET_SCHEMA,
    ASSET_TAG,
    REPOSITORY_BOUND_KEYS,
    V101_SUMMARY,
    V102_OUTPUT,
    json_bytes,
    read_json,
    relative_to_repo,
    sanitize_repo_paths,
    sha256_bytes,
    sha256_file,
)


V101_SCHEMA = "relational_teacher_v101_fullfield_supervision_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--v101-summary", type=Path, default=V101_SUMMARY)
    parser.add_argument(
        "--output", type=Path, default=Path("dist") / ASSET_NAME
    )
    return parser.parse_args()


def _add_tree(
    files: Dict[str, Path], root: Path, repo_root: Path, exclude_experiments: bool
) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("asset tree contains a symlink: " + str(path))
        if not path.is_file():
            continue
        relative_inside = path.relative_to(root)
        if exclude_experiments and "experiments" in relative_inside.parts:
            continue
        files[relative_to_repo(path, repo_root)] = path


def _sanitized_bound_json(path: Path, repo_root: Path) -> bytes:
    return json_bytes(sanitize_repo_paths(read_json(path), repo_root))


def _normalized_tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def _write_tar_member(
    archive: tarfile.TarFile, name: str, source: Union[Path, bytes]
) -> None:
    if isinstance(source, bytes):
        archive.addfile(_normalized_tar_info(name, len(source)), io.BytesIO(source))
        return
    size = source.stat().st_size
    with source.open("rb") as handle:
        archive.addfile(_normalized_tar_info(name, size), handle)


def build_bundle(repo_root: Path, summary_file: Path, output_file: Path) -> None:
    repo_root = repo_root.expanduser().resolve()
    summary_file = (
        summary_file
        if summary_file.is_absolute()
        else repo_root / summary_file
    ).expanduser().resolve()
    output_file = (
        output_file if output_file.is_absolute() else repo_root / output_file
    ).expanduser().resolve()
    if output_file.exists() or Path(str(output_file) + ".sha256").exists():
        raise FileExistsError("refusing to overwrite an existing asset bundle")
    summary = read_json(summary_file)
    if (
        summary.get("schema") != V101_SCHEMA
        or summary.get("status") != "FAIL"
        or summary.get("selected_step") is not None
        or summary.get("serialized_model_state") is not False
        or summary.get("failed_checks")
        != ["at_least_one_fullfield_candidate_passes_actual_k3"]
    ):
        raise ValueError("input is not the sealed Teacher-v10.1 failure")
    paths = summary.get("paths")
    hashes = summary.get("path_sha256")
    if (
        not isinstance(paths, Mapping)
        or not isinstance(hashes, Mapping)
        or set(paths) != set(hashes)
        or set(REPOSITORY_BOUND_KEYS) - set(paths)
        or "checkpoint" in paths
    ):
        raise ValueError("Teacher-v10.1 bound path inventory changed")

    bound_paths: Dict[str, Path] = {}
    bound_relative: Dict[str, str] = {}
    for key, raw in paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != str(hashes[key]):
            raise ValueError("Teacher-v10.1 bound file changed: " + str(key))
        bound_paths[str(key)] = path
        bound_relative[str(key)] = relative_to_repo(path, repo_root)

    dataset_root = bound_paths["dataset_index"].parent
    source_root = bound_paths["source_dataset_index"].parent
    v5_root = bound_paths["v5_checkpoint"].parents[2]
    payload_sources: Dict[str, Path] = {}
    for root in (dataset_root, source_root, v5_root):
        _add_tree(payload_sources, root, repo_root, exclude_experiments=True)

    repository_files: Dict[str, str] = {}
    overrides: Dict[str, bytes] = {}
    for key, path in bound_paths.items():
        relative = bound_relative[key]
        if key in REPOSITORY_BOUND_KEYS:
            repository_files[relative] = sha256_file(path)
            continue
        payload_sources[relative] = path
        if path.suffix.lower() == ".json":
            overrides[relative] = _sanitized_bound_json(path, repo_root)

    source_index_relative = bound_relative["source_dataset_index"]
    source_index_bytes = overrides.get(source_index_relative)
    if source_index_bytes is None:
        source_index_bytes = _sanitized_bound_json(
            bound_paths["source_dataset_index"], repo_root
        )
        overrides[source_index_relative] = source_index_bytes

    dataset_index_relative = bound_relative["dataset_index"]
    portable_index = sanitize_repo_paths(
        read_json(bound_paths["dataset_index"]), repo_root
    )
    if not isinstance(portable_index, dict):
        raise ValueError("Teacher-v9 dataset index is not an object")
    portable_index["source_dataset_root"] = relative_to_repo(source_root, repo_root)
    if "source_index_sha256" in portable_index:
        portable_index["source_index_sha256"] = sha256_bytes(source_index_bytes)
    overrides[dataset_index_relative] = json_bytes(portable_index)

    portable_summary = copy.deepcopy(summary)
    portable_summary["paths"] = dict(sorted(bound_relative.items()))
    portable_hashes: Dict[str, str] = {}
    for key, relative in bound_relative.items():
        if key in REPOSITORY_BOUND_KEYS:
            portable_hashes[key] = repository_files[relative]
        elif relative in overrides:
            portable_hashes[key] = sha256_bytes(overrides[relative])
        else:
            portable_hashes[key] = sha256_file(payload_sources[relative])
    portable_summary["path_sha256"] = portable_hashes
    portable_summary["portable_paths"] = True
    portable_summary["portable_asset_schema"] = ASSET_SCHEMA
    portable_summary_bytes = json_bytes(portable_summary)
    summary_relative = V101_SUMMARY.as_posix()
    if relative_to_repo(summary_file, repo_root) != summary_relative:
        raise ValueError("Teacher-v10.1 summary is not at its canonical path")
    overrides[summary_relative] = portable_summary_bytes
    payload_sources[summary_relative] = summary_file

    payload_hashes: Dict[str, str] = {}
    payload_sizes: Dict[str, int] = {}
    for relative, source in sorted(payload_sources.items()):
        if relative in overrides:
            payload_hashes[relative] = sha256_bytes(overrides[relative])
            payload_sizes[relative] = len(overrides[relative])
        else:
            payload_hashes[relative] = sha256_file(source)
            payload_sizes[relative] = source.stat().st_size
    manifest = {
        "schema": ASSET_SCHEMA,
        "asset_tag": ASSET_TAG,
        "v101_summary": summary_relative,
        "expected_v102_output": V102_OUTPUT.as_posix(),
        "files": payload_hashes,
        "repository_files": dict(sorted(repository_files.items())),
        "file_count": len(payload_hashes),
        "total_uncompressed_bytes": sum(payload_sizes.values()),
        "machine_paths_removed_from_bound_json": True,
    }
    manifest_bytes = json_bytes(manifest)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix="." + output_file.name + ".",
        suffix=".tmp",
        dir=str(output_file.parent),
        delete=False,
    )
    temporary = Path(temporary_handle.name)
    temporary_handle.close()
    try:
        with tarfile.open(temporary, mode="w:gz") as archive:
            for relative, source in sorted(payload_sources.items()):
                _write_tar_member(archive, relative, overrides.get(relative, source))
            _write_tar_member(archive, ASSET_MANIFEST.as_posix(), manifest_bytes)
        os.replace(temporary, output_file)
    finally:
        if temporary.exists():
            temporary.unlink()
    archive_hash = sha256_file(output_file)
    checksum_file = Path(str(output_file) + ".sha256")
    checksum_file.write_text(
        archive_hash + "  " + output_file.name + "\n", encoding="utf-8"
    )
    print("[ASSET_PACKAGE_PASS] Teacher-v10.2 portable prerequisites")
    print("[OK] files:", len(payload_hashes))
    print("[OK] uncompressed bytes:", sum(payload_sizes.values()))
    print("[OK] archive:", output_file)
    print("[OK] sha256:", checksum_file)


def main() -> None:
    args = _parse_args()
    build_bundle(args.repo_root, args.v101_summary, args.output)


if __name__ == "__main__":
    main()

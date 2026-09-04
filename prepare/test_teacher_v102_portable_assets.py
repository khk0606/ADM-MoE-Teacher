#!/usr/bin/env python3
"""CPU contract for the Teacher-v10.2 portable asset transport."""

from __future__ import annotations

import io
import json
import tarfile
import tempfile
from pathlib import Path

from teacher_v102_portable_assets import (
    ASSET_MANIFEST,
    ASSET_SCHEMA,
    ASSET_TAG,
    V101_SUMMARY,
    V102_OUTPUT,
    extract_verified_archive,
    json_bytes,
    resolve_repo_path,
    sanitize_repo_paths,
    sha256_bytes,
    validate_installed_assets,
)


def _add_bytes(archive: tarfile.TarFile, name: str, value: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(value)
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(value))


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="teacher-v102-portable-test-") as raw:
        root = Path(raw) / "repo"
        root.mkdir()
        source_path = root / "prepare/source.py"
        source_path.parent.mkdir(parents=True)
        source_path.write_bytes(b"SOURCE\n")
        summary_bytes = b'{"portable": true}\n'
        payload_bytes = b"PAYLOAD\n"
        manifest = {
            "schema": ASSET_SCHEMA,
            "asset_tag": ASSET_TAG,
            "v101_summary": V101_SUMMARY.as_posix(),
            "files": {
                V101_SUMMARY.as_posix(): sha256_bytes(summary_bytes),
                "data/payload.bin": sha256_bytes(payload_bytes),
            },
            "repository_files": {
                "prepare/source.py": sha256_bytes(b"SOURCE\n"),
            },
            "file_count": 2,
            "expected_v102_output": V102_OUTPUT.as_posix(),
            "machine_paths_removed_from_bound_json": True,
        }
        manifest_bytes = json_bytes(manifest)
        archive_file = Path(raw) / "assets.tar.gz"
        with tarfile.open(archive_file, "w:gz") as archive:
            _add_bytes(archive, V101_SUMMARY.as_posix(), summary_bytes)
            _add_bytes(archive, "data/payload.bin", payload_bytes)
            _add_bytes(archive, ASSET_MANIFEST.as_posix(), manifest_bytes)
        extract_verified_archive(archive_file, root)
        validated = validate_installed_assets(root)
        if validated["schema"] != ASSET_SCHEMA:
            raise AssertionError("installed asset schema changed")
        if (root / "data/payload.bin").read_bytes() != payload_bytes:
            raise AssertionError("portable payload changed during extraction")

        machine_root = Path("/home/example/AMDM")
        sanitized = sanitize_repo_paths(
            {"a": "/home/example/AMDM/data/value.bin", "b": "unchanged"},
            machine_root,
        )
        if sanitized != {"a": "data/value.bin", "b": "unchanged"}:
            raise AssertionError("machine path was not sanitized")
        try:
            resolve_repo_path(root, "../escape")
        except ValueError:
            pass
        else:
            raise AssertionError("path traversal was accepted")

        conflict_root = Path(raw) / "conflict"
        conflict_root.mkdir()
        conflict_source = conflict_root / "prepare/source.py"
        conflict_source.parent.mkdir(parents=True)
        conflict_source.write_bytes(b"SOURCE\n")
        conflict_payload = conflict_root / "data/payload.bin"
        conflict_payload.parent.mkdir(parents=True)
        conflict_payload.write_bytes(b"DIFFERENT\n")
        try:
            extract_verified_archive(archive_file, conflict_root)
        except FileExistsError:
            pass
        else:
            raise AssertionError("a conflicting local asset was overwritten")

    print("[PASS] Teacher-v10.2 portable asset CPU contract")
    print("[PASS] verified extraction, source binding and path traversal guards")
    print("[PASS] conflicting local files fail closed without overwrite")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Portable asset contracts for the public Teacher-v10.2 reproduction."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import BinaryIO, Dict, List, Mapping, MutableMapping, Optional, Tuple


ASSET_SCHEMA = "teacher_v102_portable_assets_v1"
ASSET_TAG = "teacher-v102-assets-v1"
ASSET_NAME = "teacher-v102-assets-v1.tar.gz"
ASSET_MANIFEST = Path("data/teacher_v102_assets_manifest.json")
DEFAULT_ASSET_URL = (
    "https://github.com/khk0606/ADM-MoE-Teacher/releases/download/"
    + ASSET_TAG
    + "/"
    + ASSET_NAME
)
V101_SUMMARY = Path(
    "data/history_affordance_relational_teacher_v9_all_sittable_v1/"
    "experiments/teacher_lora_v101/"
    "fullfield_supervision_s20261030_v1/summary.json"
)
V102_OUTPUT = Path(
    "data/history_affordance_relational_teacher_v9_all_sittable_v1/"
    "experiments/teacher_lora_v102/"
    "dense_instance_supervision_s20261031_v1"
)
REPOSITORY_BOUND_KEYS = frozenset({"runner", "validator", "contract", "objective"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def read_json(path: Path) -> MutableMapping[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: " + str(path))
    return value


def json_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_repo_path(repo_root: Path, raw: object) -> Path:
    repo_root = Path(repo_root).expanduser().resolve()
    relative = Path(str(raw))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("asset path must be repository-relative: " + str(raw))
    result = (repo_root / relative).resolve()
    try:
        result.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("asset path escapes repository: " + str(raw)) from exc
    return result


def relative_to_repo(path: Path, repo_root: Path) -> str:
    path = Path(path).expanduser().resolve()
    repo_root = Path(repo_root).expanduser().resolve()
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise ValueError("bound file is outside the repository: " + str(path)) from exc


def sanitize_repo_paths(value: object, repo_root: Path) -> object:
    """Replace exact machine-local repository paths with relative paths."""

    repo_root = Path(repo_root).expanduser()
    if not repo_root.is_absolute():
        repo_root = repo_root.absolute()
    prefix = str(repo_root)
    if isinstance(value, Mapping):
        return {
            str(key): sanitize_repo_paths(nested, repo_root)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [sanitize_repo_paths(nested, repo_root) for nested in value]
    if isinstance(value, str):
        if value == prefix:
            return "."
        if value.startswith(prefix + os.sep):
            return Path(value).relative_to(repo_root).as_posix()
    return value


def _validated_hash_map(value: object, label: str) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(label + " must be an object")
    result = {str(name): str(digest) for name, digest in value.items()}
    if any(not _SHA256_RE.fullmatch(digest) for digest in result.values()):
        raise ValueError(label + " contains an invalid SHA-256")
    return result


def _validate_inventory_locations(
    payload: Mapping[str, str], repository: Mapping[str, str]
) -> None:
    for name in payload:
        parts = Path(name).parts
        if not parts or parts[0] not in {"data", "outputs"}:
            raise ValueError("asset payload may only target data/ or outputs/: " + name)
    for name in repository:
        parts = Path(name).parts
        if not parts or parts[0] != "prepare":
            raise ValueError("repository binding may only target prepare/: " + name)


def validate_installed_assets(repo_root: Path) -> Mapping[str, object]:
    repo_root = Path(repo_root).expanduser().resolve()
    manifest_file = repo_root / ASSET_MANIFEST
    if not manifest_file.is_file():
        raise FileNotFoundError("missing Teacher-v10.2 asset manifest: " + str(manifest_file))
    manifest = read_json(manifest_file)
    if (
        manifest.get("schema") != ASSET_SCHEMA
        or manifest.get("asset_tag") != ASSET_TAG
        or manifest.get("v101_summary") != V101_SUMMARY.as_posix()
    ):
        raise ValueError("Teacher-v10.2 asset manifest identity changed")
    payload = _validated_hash_map(manifest.get("files"), "asset files")
    repository = _validated_hash_map(
        manifest.get("repository_files"), "repository files"
    )
    _validate_inventory_locations(payload, repository)
    if not payload or not repository or set(payload) & set(repository):
        raise ValueError("Teacher-v10.2 asset inventory is empty or overlaps source")
    if (
        manifest.get("file_count") != len(payload)
        or manifest.get("expected_v102_output") != V102_OUTPUT.as_posix()
        or manifest.get("machine_paths_removed_from_bound_json") is not True
    ):
        raise ValueError("Teacher-v10.2 asset manifest metadata changed")
    if V101_SUMMARY.as_posix() not in payload:
        raise ValueError("portable Teacher-v10.1 summary is absent")
    for inventory_name, inventory in (("asset", payload), ("source", repository)):
        for name, expected in inventory.items():
            path = resolve_repo_path(repo_root, name)
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(
                    "Teacher-v10.2 {} file missing or changed: {}".format(
                        inventory_name, name
                    )
                )
    return manifest


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ADM-MoE-Teacher-v10.2-bootstrap"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)


def _manifest_from_archive(
    archive: tarfile.TarFile,
) -> Tuple[Mapping[str, object], bytes]:
    names = [member.name for member in archive.getmembers()]
    if len(names) != len(set(names)):
        raise ValueError("asset archive contains duplicate paths")
    manifest_name = ASSET_MANIFEST.as_posix()
    try:
        member = archive.getmember(manifest_name)
    except KeyError as exc:
        raise ValueError("asset archive has no manifest") from exc
    if not member.isfile():
        raise ValueError("asset manifest is not a regular file")
    source = archive.extractfile(member)
    if source is None:
        raise ValueError("asset manifest cannot be read")
    raw = source.read()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("asset manifest is not a JSON object")
    return value, raw


def _copy_member(source: BinaryIO, destination: Path, expected: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix="." + destination.name + ".",
        suffix=".tmp",
        dir=str(destination.parent),
        delete=False,
    )
    temporary = Path(handle.name)
    digest = hashlib.sha256()
    try:
        with handle:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
                digest.update(block)
            handle.flush()
            os.fsync(handle.fileno())
        if digest.hexdigest() != expected:
            raise ValueError("downloaded asset hash changed: " + str(destination))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def extract_verified_archive(archive_file: Path, repo_root: Path) -> None:
    repo_root = Path(repo_root).expanduser().resolve()
    created: List[Path] = []
    with tarfile.open(archive_file, mode="r:gz") as archive:
        manifest, manifest_raw = _manifest_from_archive(archive)
        if (
            manifest.get("schema") != ASSET_SCHEMA
            or manifest.get("asset_tag") != ASSET_TAG
            or manifest.get("v101_summary") != V101_SUMMARY.as_posix()
        ):
            raise ValueError("downloaded Teacher-v10.2 asset identity changed")
        files = _validated_hash_map(manifest.get("files"), "asset files")
        repository = _validated_hash_map(
            manifest.get("repository_files"), "repository files"
        )
        _validate_inventory_locations(files, repository)
        expected_members = set(files) | {ASSET_MANIFEST.as_posix()}
        actual_members = set()
        for member in archive.getmembers():
            if not member.isfile():
                raise ValueError("asset archive contains a non-file member: " + member.name)
            resolve_repo_path(repo_root, member.name)
            actual_members.add(member.name)
        if actual_members != expected_members:
            raise ValueError("asset archive inventory differs from its manifest")
        for name, expected in repository.items():
            source_path = resolve_repo_path(repo_root, name)
            if not source_path.is_file() or sha256_file(source_path) != expected:
                raise ValueError("clone source does not match asset bundle: " + name)
        manifest_path = resolve_repo_path(repo_root, ASSET_MANIFEST)
        targets = {name: resolve_repo_path(repo_root, name) for name in files}
        for name, target in targets.items():
            if target.exists() and (
                not target.is_file() or sha256_file(target) != files[name]
            ):
                raise FileExistsError("refusing to overwrite a different file: " + name)
        if manifest_path.exists() and (
            not manifest_path.is_file()
            or manifest_path.read_bytes() != manifest_raw
        ):
            raise FileExistsError("refusing to overwrite a different asset manifest")
        try:
            for name, target in targets.items():
                if target.exists():
                    continue
                source = archive.extractfile(archive.getmember(name))
                if source is None:
                    raise ValueError("asset member cannot be read: " + name)
                _copy_member(source, target, files[name])
                created.append(target)
            if not manifest_path.exists():
                _copy_member(
                    source=io.BytesIO(manifest_raw),
                    destination=manifest_path,
                    expected=sha256_bytes(manifest_raw),
                )
                created.append(manifest_path)
            validate_installed_assets(repo_root)
        except Exception:
            for path in reversed(created):
                path.unlink(missing_ok=True)
            raise


def fetch_and_install_assets(
    repo_root: Path,
    asset_url: str = DEFAULT_ASSET_URL,
    checksum_url: Optional[str] = None,
) -> Mapping[str, object]:
    repo_root = Path(repo_root).expanduser().resolve()
    try:
        return validate_installed_assets(repo_root)
    except FileNotFoundError:
        pass
    checksum_url = checksum_url or asset_url + ".sha256"
    with tempfile.TemporaryDirectory(prefix="teacher-v102-assets-") as raw_temp:
        temp = Path(raw_temp)
        archive_file = temp / ASSET_NAME
        checksum_file = temp / (ASSET_NAME + ".sha256")
        _download(checksum_url, checksum_file)
        checksum_tokens = checksum_file.read_text(encoding="utf-8").strip().split()
        if not checksum_tokens or not _SHA256_RE.fullmatch(checksum_tokens[0]):
            raise ValueError("downloaded checksum file is invalid")
        _download(asset_url, archive_file)
        actual = sha256_file(archive_file)
        if actual != checksum_tokens[0]:
            raise ValueError("Teacher-v10.2 archive checksum mismatch")
        extract_verified_archive(archive_file, repo_root)
    return validate_installed_assets(repo_root)


__all__ = [
    "ASSET_MANIFEST",
    "ASSET_NAME",
    "ASSET_SCHEMA",
    "ASSET_TAG",
    "DEFAULT_ASSET_URL",
    "REPOSITORY_BOUND_KEYS",
    "V101_SUMMARY",
    "V102_OUTPUT",
    "extract_verified_archive",
    "fetch_and_install_assets",
    "json_bytes",
    "read_json",
    "relative_to_repo",
    "resolve_repo_path",
    "sanitize_repo_paths",
    "sha256_bytes",
    "sha256_file",
    "validate_installed_assets",
]

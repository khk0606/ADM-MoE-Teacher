#!/usr/bin/env python3
"""Download, verify, and install the public Teacher-v10.2 asset bundle."""

from __future__ import annotations

import argparse
import os
import urllib.error
from pathlib import Path

from teacher_v102_portable_assets import (
    DEFAULT_ASSET_URL,
    fetch_and_install_assets,
    validate_installed_assets,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--asset-url",
        default=os.environ.get("TEACHER_V102_ASSET_URL", DEFAULT_ASSET_URL),
    )
    parser.add_argument(
        "--checksum-url",
        default=os.environ.get("TEACHER_V102_ASSET_CHECKSUM_URL"),
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.check_only:
            manifest = validate_installed_assets(args.repo_root)
        else:
            manifest = fetch_and_install_assets(
                args.repo_root,
                asset_url=args.asset_url,
                checksum_url=args.checksum_url,
            )
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            "[STOP] Teacher-v10.2 Release asset is unavailable (HTTP {}): {}".format(
                exc.code, args.asset_url
            )
        ) from exc
    except urllib.error.URLError as exc:
        raise SystemExit(
            "[STOP] cannot download the Teacher-v10.2 Release asset: {}".format(
                exc.reason
            )
        ) from exc
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        raise SystemExit("[STOP] Teacher-v10.2 asset validation failed: " + str(exc)) from exc
    print("[ASSET_PASS] Teacher-v10.2 portable prerequisites")
    print("[OK] files:", len(manifest["files"]))
    print("[OK] source bindings:", len(manifest["repository_files"]))


if __name__ == "__main__":
    main()

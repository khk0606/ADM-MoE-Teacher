#!/usr/bin/env python3
"""Revalidate a promoted Base-teacher sample/cache index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PREPARE_DIR = Path(__file__).resolve().parent
if str(PREPARE_DIR) not in sys.path:
    sys.path.insert(0, str(PREPARE_DIR))

from build_base_teacher_cache_index import validate_cache_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_cache_index(args.index), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
cd "$ROOT"

python prepare/fetch_teacher_v102_assets.py --repo-root "$ROOT"

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
cd "$ROOT"
sha256sum --quiet -c SMALL_ROOM30_TARGETED_V4_SHA256SUMS.txt
sha256sum --quiet -c SMALL_ROOM30_TARGETED_V4_DEPENDENCIES_SHA256SUMS.txt
python -u prepare/view_small_room30_candidates_v4.py --port "${PORT:-8092}" \
  --run-summary "${SUMMARY:-outputs/small_room30_targeted_v4_run01/summary.json}" "$@"

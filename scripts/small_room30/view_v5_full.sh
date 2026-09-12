#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
cd "$ROOT"
sha256sum --quiet -c SMALL_ROOM30_V5_FULL_SHA256SUMS.txt
sha256sum --quiet -c SMALL_ROOM30_V5_FULL_DEPENDENCIES_SHA256SUMS.txt
python -u prepare/view_small_room30_evaluation.py \
  --summary outputs/small_room30_targeted_v5_eval_full01/summary.json --port "${PORT:-8092}" "$@"

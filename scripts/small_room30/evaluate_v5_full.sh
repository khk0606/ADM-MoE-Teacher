#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
cd "$ROOT"
if [[ $# -ne 0 ]]; then echo 'Standard full evaluation takes no arguments.' >&2; exit 2; fi
sha256sum --quiet -c SMALL_ROOM30_V5_FULL_SHA256SUMS.txt
sha256sum --quiet -c SMALL_ROOM30_V5_FULL_DEPENDENCIES_SHA256SUMS.txt
python -u prepare/test_small_room30_v5_full.py
EVAL_OUT=outputs/small_room30_targeted_v5_eval_full01
mkdir -p outputs
if [[ ! -f "$EVAL_OUT/summary.json" ]]; then
  python -u prepare/evaluate_small_room30_v5_full.py --mode full --resume \
    --output-dir "$EVAL_OUT" 2>&1 | tee -a "${EVAL_OUT}.console.log"
fi
python -u prepare/review_small_room30_v5_full.py
echo '[VIEW] bash scripts/small_room30/view_v5_full.sh'

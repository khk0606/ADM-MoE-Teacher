#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
cd "$ROOT"
sha256sum --quiet -c SMALL_ROOM30_TARGETED_V5_SHA256SUMS.txt
sha256sum --quiet -c SMALL_ROOM30_TARGETED_V5_DEPENDENCIES_SHA256SUMS.txt
OUT="${OUT:-outputs/small_room30_targeted_v5_run01}"
if [[ -e "$OUT" || -e "${OUT}.console.log" ]]; then
  echo "[STOP] Existing run/log preserved: $OUT. Do not restart; send status." >&2; exit 2
fi
mkdir -p "$(dirname "$OUT")"
python -u prepare/run_small_room30_targeted_v5.py --mode train --allow-research-training \
  --readiness "${READINESS:-outputs/small_room30_targeted_v5_ready01/summary.json}" \
  --output-dir "$OUT" "$@" 2>&1 | tee "${OUT}.console.log"
echo '[STOP] Bounded pilot ended. Inspect saved candidates in Viser; do not extend old runs.'

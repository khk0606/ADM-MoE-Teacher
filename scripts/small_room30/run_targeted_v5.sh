#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
cd "$ROOT"
if [[ $# -ne 0 ]]; then echo 'Use separate prepare/train scripts for explicit path overrides.' >&2; exit 2; fi
if [[ -n "${OUT:-}" || -n "${READINESS:-}" ]]; then echo 'Unset OUT/READINESS for the standard one-shot run.' >&2; exit 2; fi
bash scripts/small_room30/prepare_targeted_v5.sh
bash scripts/small_room30/train_targeted_v5.sh
python prepare/review_small_room30_targeted_v5.py --zip small_room30_targeted_v5_review01.zip
echo '[NEXT] bash scripts/small_room30/view_candidates_v5.sh'

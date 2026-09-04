#!/usr/bin/env bash
set -euo pipefail
trap 'BOOTSTRAP_RC=$?; echo "[DONE] Teacher-v10.2 bootstrap exit=$BOOTSTRAP_RC"' EXIT

ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
cd "$ROOT"

python -c 'import numpy; print("[PASS] Python/NumPy available for asset validation")'
bash scripts/teacher_v102/fetch_assets.sh
python prepare/fetch_teacher_v102_assets.py --repo-root "$ROOT" --check-only

FAILED="data/history_affordance_relational_teacher_v9_all_sittable_v1/experiments/teacher_lora_v101/fullfield_supervision_s20261030_v1/summary.json"
python prepare/validate_relational_teacher_v101_fullfield_supervision.py \
  --summary "$FAILED"

echo "[ASSET_READY] Teacher-v10.2 data and checkpoints are installed and verified"
echo "[NEXT] bash scripts/teacher_v102/reproduce.sh"

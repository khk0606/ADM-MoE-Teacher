#!/usr/bin/env bash
set -euo pipefail
trap 'VIEW_RC=$?; echo "[DONE] Teacher-v10.2 Viser exit=$VIEW_RC"' EXIT

ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
cd "$ROOT"

SUMMARY="data/history_affordance_relational_teacher_v9_all_sittable_v1/experiments/teacher_lora_v102/dense_instance_supervision_s20261031_v1/summary.json"
if [[ ! -f "$SUMMARY" ]]; then
  echo "[STOP] run bash scripts/teacher_v102/reproduce.sh first"
  exit 1
fi

python prepare/validate_relational_teacher_v102_dense_instance_supervision.py --summary "$SUMMARY"
python -c 'import numpy, viser; print("[PASS] numpy/viser import")'
echo "[OPEN] http://localhost:8080"
python prepare/visualize_relational_teacher_v102_dense_instance_viser.py --summary "$SUMMARY" --host 0.0.0.0 --port 8080

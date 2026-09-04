#!/usr/bin/env bash
set -euo pipefail
trap 'VIEW_RC=$?; echo "[DONE] Teacher-v10.2 viewer exit=$VIEW_RC; this terminal remains open"' EXIT

AMDM_ROOT="$(pwd -P)"
PACKAGE_ROOT="$(cd "$(dirname "$0")" && pwd -P)"
PATCH="$PACKAGE_ROOT"

python "$PATCH/prepare/validate_teacher_lora_v102_dense_instance_package.py"
cp "$PATCH"/prepare/*.py "$AMDM_ROOT/prepare/"
cd "$AMDM_ROOT"
DATA=data/history_affordance_relational_teacher_v9_all_sittable_v1
SUMMARY="$DATA/experiments/teacher_lora_v102/dense_instance_supervision_s20261031_v1/summary.json"
test -f "$SUMMARY" || { echo "[STOP] missing Teacher-v10.2 summary: $SUMMARY"; exit 1; }
python prepare/validate_relational_teacher_v102_dense_instance_supervision.py --summary "$SUMMARY"
python -c 'import viser; print("[PASS] viser import")'
python prepare/visualize_relational_teacher_v102_dense_instance_viser.py --summary "$SUMMARY" --host 0.0.0.0 --port 8080

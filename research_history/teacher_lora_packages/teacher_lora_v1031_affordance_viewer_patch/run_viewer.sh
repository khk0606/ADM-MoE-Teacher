#!/usr/bin/env bash
set -euo pipefail
trap 'VIEW_RC=$?; echo "[DONE] Teacher-v10.3.1 viewer exit=$VIEW_RC; this terminal remains open"' EXIT

AMDM_ROOT="$(pwd -P)"
PACKAGE_ROOT="$(cd "$(dirname "$0")" && pwd -P)"
SUMMARY="$AMDM_ROOT/data/history_affordance_relational_teacher_v9_all_sittable_v1/experiments/teacher_lora_v1031/onpolicy_calibration6_s20261102_v3/summary.json"
VALIDATOR="$AMDM_ROOT/prepare/validate_relational_teacher_v1031_onpolicy_calibration6.py"

test -f "$SUMMARY" || {
  echo "[STOP] missing Teacher-v10.3.1 PASS summary: $SUMMARY"
  exit 1
}
test -f "$VALIDATOR" || {
  echo "[STOP] missing fixed Teacher-v10.3.1 validator: $VALIDATOR"
  exit 1
}

python "$PACKAGE_ROOT/prepare/validate_teacher_lora_v1031_affordance_viewer_package.py"
PYTHONPATH="$PACKAGE_ROOT/prepare" python "$PACKAGE_ROOT/prepare/test_relational_teacher_v1031_affordance_viewer_common.py"
python -c 'import numpy, viser; print("[PASS] numpy and viser imports")'
python "$VALIDATOR" --summary "$SUMMARY"

echo "[OPEN] http://localhost:8080"
echo "[INFO] stop only the viewer with Ctrl+C; the terminal stays open"
PYTHONPATH="$PACKAGE_ROOT/prepare" python -u "$PACKAGE_ROOT/prepare/visualize_relational_teacher_v1031_affordance_viser.py" --summary "$SUMMARY" --host 0.0.0.0 --port 8080

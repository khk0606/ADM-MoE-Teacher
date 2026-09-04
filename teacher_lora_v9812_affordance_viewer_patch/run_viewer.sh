#!/usr/bin/env bash
set -euo pipefail

AMDM_ROOT="$(pwd -P)"
PACKAGE_ROOT="$(cd "$(dirname "$0")" && pwd -P)"
SUMMARY="$AMDM_ROOT/data/history_affordance_relational_teacher_v9_all_sittable_v1/experiments/teacher_lora_v9812/two_scene_multiupdate_calibration_s20261028_v1/summary.json"

test -f "$SUMMARY" || {
  echo "[STOP] missing Teacher-v9.8.12 summary: $SUMMARY"
  exit 1
}

python "$PACKAGE_ROOT/prepare/validate_teacher_lora_v9812_affordance_viewer_package.py"
PYTHONPATH="$PACKAGE_ROOT/prepare" python "$PACKAGE_ROOT/prepare/test_relational_teacher_v9812_affordance_viewer_common.py"
python -c 'import numpy, viser; print("[PASS] numpy and viser imports")'

echo "[OPEN] http://localhost:8080"
echo "[INFO] stop only the viewer with Ctrl+C; the terminal stays open"
PYTHONPATH="$PACKAGE_ROOT/prepare" python -u "$PACKAGE_ROOT/prepare/visualize_relational_teacher_v9812_affordance_viser.py" --summary "$SUMMARY" --host 0.0.0.0 --port 8080

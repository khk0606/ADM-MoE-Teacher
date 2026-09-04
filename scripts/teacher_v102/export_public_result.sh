#!/usr/bin/env bash
set -euo pipefail
trap 'RUN_RC=$?; echo "[DONE] Teacher-v10.2 public export exit=$RUN_RC; this terminal remains open"' EXIT

ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
cd "$ROOT"

SUMMARY="${1:-data/history_affordance_relational_teacher_v9_all_sittable_v1/experiments/teacher_lora_v102/dense_instance_supervision_s20261031_v1/summary.json}"
OUTPUT_DIR="${2:-teacher_v102_public_bundle}"

python prepare/validate_relational_teacher_v102_dense_instance_supervision.py --summary "$SUMMARY"
python prepare/export_relational_teacher_v102_public_bundle.py --summary "$SUMMARY" --output-dir "$OUTPUT_DIR"
python -c 'import numpy, viser; print("[PASS] numpy/viser import")'

echo "[OPEN] http://localhost:8080"
python prepare/visualize_relational_teacher_v102_public_viser.py --bundle-dir "$OUTPUT_DIR" --host 0.0.0.0 --port 8080

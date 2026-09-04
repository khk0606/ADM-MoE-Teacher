#!/usr/bin/env bash
set -euo pipefail
trap 'RUN_RC=$?; echo "[DONE] Teacher-v10.3.1 block exit=$RUN_RC; this terminal remains open"' EXIT

AMDM_ROOT="$(pwd -P)"
PACKAGE_ROOT="$(cd "$(dirname "$0")" && pwd -P)"
PATCH="$PACKAGE_ROOT"

python "$PATCH/prepare/validate_teacher_lora_v1031_onpolicy_calibration6_package.py"

MISSING=0
for FILE in fewshot_cdm_common.py fewshot_cdm_lora.py relational_teacher_v9_all_sittable_contract.py relational_teacher_v9_all_sittable_metrics.py relational_teacher_v9_lora_objective.py relational_teacher_v9_lora_preflight_contract.py relational_teacher_v9_lora_runtime.py relational_teacher_v94_common_descent.py relational_teacher_v10_supervised_capacity_contract.py relational_teacher_v10_supervised_objective.py run_relational_teacher_v10_supervised_capacity.py validate_relational_teacher_v10_supervised_capacity.py relational_teacher_v101_fullfield_contract.py relational_teacher_v102_dense_instance_contract.py relational_teacher_v102_dense_instance_objective.py relational_teacher_v103_onpolicy_response_contract.py preflight_relational_teacher_v103_onpolicy_response.py validate_relational_teacher_v103_onpolicy_response.py train_fewshot_cdm.py; do
  if [ ! -f "$AMDM_ROOT/prepare/$FILE" ]; then
    echo "[STOP] missing prepare/$FILE"
    MISSING=1
  fi
done
if [ "$MISSING" -ne 0 ]; then
  echo "[STOP] dependencies are incomplete; do not run CUDA"
  exit 1
fi

cp "$PATCH"/prepare/*.py "$AMDM_ROOT/prepare/"
cd "$AMDM_ROOT"
python prepare/test_relational_teacher_v1031_onpolicy_calibration6_contract.py

DATA=data/history_affordance_relational_teacher_v9_all_sittable_v1
V103="$DATA/experiments/teacher_lora_v103/onpolicy_response_s20261101_v1/preflight.json"
OUT="$DATA/experiments/teacher_lora_v1031/onpolicy_calibration6_s20261102_v3"
test -f "$V103" || { echo "[STOP] missing Teacher-v10.3 PASS: $V103"; exit 1; }

python prepare/validate_relational_teacher_v103_onpolicy_response.py --report "$V103"
if [ -f "$OUT/summary.json" ]; then
  echo "[REUSE] completed v3 summary; validating without CUDA rerun"
  python prepare/validate_relational_teacher_v1031_onpolicy_calibration6.py --summary "$OUT/summary.json"
  python prepare/summarize_relational_teacher_v1031_onpolicy_calibration6.py --summary "$OUT/summary.json"
  exit 0
fi
test ! -e "$OUT" || { echo "[STOP] incomplete v3 output exists; preserve it for diagnosis"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ARGS=(--v103-report "$V103" --output-dir "$OUT" --device cuda:0 --no-progress)
python -u prepare/run_relational_teacher_v1031_onpolicy_calibration6.py "${ARGS[@]}"
python prepare/validate_relational_teacher_v1031_onpolicy_calibration6.py --summary "$OUT/summary.json"
python prepare/summarize_relational_teacher_v1031_onpolicy_calibration6.py --summary "$OUT/summary.json"

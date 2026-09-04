#!/usr/bin/env bash
set -euo pipefail
trap 'RUN_RC=$?; echo "[DONE] Teacher-v10.1 block exit=$RUN_RC; this terminal remains open"' EXIT

AMDM_ROOT="$(pwd -P)"
PACKAGE_ROOT="$(cd "$(dirname "$0")" && pwd -P)"
PATCH="$PACKAGE_ROOT"

python "$PATCH/prepare/validate_teacher_lora_v101_fullfield_package.py"

MISSING=0
for FILE in fewshot_cdm_common.py fewshot_cdm_lora.py relational_teacher_v9_all_sittable_contract.py relational_teacher_v9_all_sittable_metrics.py relational_teacher_v9_lora_objective.py relational_teacher_v9_lora_preflight_contract.py relational_teacher_v9_lora_runtime.py relational_teacher_v10_supervised_capacity_contract.py relational_teacher_v10_supervised_objective.py run_relational_teacher_v10_supervised_capacity.py validate_relational_teacher_v10_supervised_capacity.py relational_teacher_v10_affordance_viewer_common.py visualize_relational_teacher_v10_supervised_capacity_viser.py train_fewshot_cdm.py; do
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
python prepare/test_relational_teacher_v101_fullfield_contract.py

DATA=data/history_affordance_relational_teacher_v9_all_sittable_v1
FAILED="$DATA/experiments/teacher_lora_v10/supervised_capacity_s20261029_v1/summary.json"
OUT="$DATA/experiments/teacher_lora_v101/fullfield_supervision_s20261030_v1"
test -f "$FAILED" || { echo "[STOP] missing Teacher-v10 failure: $FAILED"; exit 1; }
test ! -e "$OUT" || { echo "[STOP] output already exists; preserve it and use a new versioned OUT"; exit 1; }

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ARGS=(--failed-v10-summary "$FAILED" --output-dir "$OUT" --device cuda:0 --no-progress)
python -u prepare/run_relational_teacher_v101_fullfield_supervision.py "${ARGS[@]}"
python prepare/validate_relational_teacher_v101_fullfield_supervision.py --summary "$OUT/summary.json"
python prepare/summarize_relational_teacher_v101_fullfield_supervision.py --summary "$OUT/summary.json"
